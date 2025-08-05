using System.Diagnostics;
using System.Globalization;
using app.Services;
using Common.Helpers;
using Common.Models.AI;
using Common.Models.Responses;
using Common.Services;
using CsvHelper;
using Dapr;
using Dapr.Client;
using DataExporter.Services;
using Microsoft.AspNetCore.Mvc;

namespace InfluxController.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class InfluxController : ControllerBase
    {
        private readonly string queryString = "import \"experimental\"" +
                                              " from(bucket: \"home_assistant\")" +
                                              " |> range(start:experimental.subDuration(d: 24h, from: today()), stop: today())" +
                                              " |> filter(fn: (r) => r[\"entity_id\"] == \"forecast_home_2\" or r[\"entity_id\"] == \"warmtepomp_power\" or r[\"entity_id\"] == \"smoke_detector_device_17_temperature\")" +
                                              " |> filter(fn: (r) =>  r[\"_field\"] == \"humidity\" or r[\"_field\"] == \"pressure\" or r[\"_field\"] == \"value\")" +
                                              " |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)" +
                                              " |> fill(usePrevious: true)" +
                                              " |> keep(columns: [\"_field\", \"_measurement\", \"_value\", \"_time\"])" +
                                              " |> pivot(" +
                                              "     rowKey: [\"_time\"], columnKey: [\"_measurement\", \"_field\"], valueColumn: \"_value\")" +
                                              " |> yield(name: \"values\")";

        private readonly IInfluxDbService _influxDbService;
        private readonly string _org;
        private readonly IFileService _fileService;
        private readonly ILogger<InfluxController> _logger;
        private readonly DaprClient _daprClient;
        private readonly ICleanupService _cleanupService;
        private readonly ILocalFileService _localFileService;
        private readonly SemaphoreSlim _semaphoreSlim = new(1, 1);

        public InfluxController(IInfluxDbService influxDBService,
            IFileService fileService,
            IConfiguration configuration,
            ICleanupService cleanupService,
            DaprClient daprClient,
            ILocalFileService localFileService,
            ILogger<InfluxController> logger)
        {
            this._influxDbService = influxDBService;
            _org = configuration.GetValue<string>("InfluxDB:Org")!;
            _fileService = fileService;
            _logger = logger;
            _daprClient = daprClient;
            _cleanupService = cleanupService;
            _localFileService = localFileService;
        }

        [HttpGet("ExportDataForDate")]
        public async Task<IActionResult> ExportDataForDate(DateTime startDate, CancellationToken token)
        {
            var strDateTime = startDate.ToStartDayString();
            _logger.LogDebug("Start export data for date {startDate}", strDateTime);
            var strStartDate = startDate.AddDays(-1).ToStartDayString();
            var strTomorrowStartDate = startDate.ToStartDayString();

            var q = "import \"experimental\"" +
                    " from(bucket: \"home_assistant\")" +
                    " |> range(start: " + strStartDate + ", stop: " + strTomorrowStartDate + ")" +
                    " |> filter(fn: (r) => r[\"entity_id\"] == \"forecast_home_2\" or r[\"entity_id\"] == \"warmtepomp_power\" or r[\"entity_id\"] == \"smoke_detector_device_17_temperature\")" +
                    " |> filter(fn: (r) =>  r[\"_field\"] == \"humidity\" or r[\"_field\"] == \"pressure\" or r[\"_field\"] == \"value\")" +
                    " |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)" +
                    " |> fill(usePrevious: true)" +
                    " |> keep(columns: [\"_field\", \"_measurement\", \"_value\", \"_time\"])" +
                    " |> pivot(" +
                    "     rowKey: [\"_time\"], columnKey: [\"_measurement\", \"_field\"], valueColumn: \"_value\")" +
                    " |> yield(name: \"values\")";

            var response = await _influxDbService.QueryAsync(q, _org, token);
            var fileName = await _fileService.RetrieveParsedFile(
                $"export-{startDate.AddDays(-1).ToString("yyyy-MM-dd")}.csv", StorageHelpers.ContainerName);
            var records = _localFileService.ReadFromFile(fileName);
            var cleanedUpResponses = _cleanupService.Cleanup(response.ToList(), records);

            var currentDate = startDate.ToString("yyyy-MM-dd");
            var generatedFileName = $"export-{currentDate}.csv";
            await _localFileService.WriteToFile(generatedFileName, cleanedUpResponses, token);
            await _fileService.UploadToAzure(StorageHelpers.ContainerName, generatedFileName, token);

            _logger.LogDebug($"Finished uploading to Azure");
            return Ok();
        }

        [Topic(pubsubName: NameConsts.INFLUX_PUBSUB_NAME, name: NameConsts.INFLUX_RETRIEVE_DATA)]
        [HttpPost(NameConsts.INFLUX_RETRIEVE_DATA)]
        public async Task RetrieveData(StartDownloadDataEvent evt, CancellationToken token)
        {
            _logger.LogInformation("Trigger received to retrieve data from influx");
            _logger.LogInformation("Current activity id: {ActivityId}", System.Diagnostics.Activity.Current?.Id);
            _logger.LogInformation("Current activity parent id: {ParentId}", Activity.Current?.ParentId);
            _logger.LogInformation("Received event with traceParent {TraceParent}", evt.TraceParent);


            if (!string.IsNullOrEmpty(evt.TraceParent))
            {
                _logger.LogInformation("Setting parent id for current activity to {TraceParent}", evt.TraceParent);
                Activity.Current?.SetParentId(evt.TraceParent);
            }

            await _semaphoreSlim.WaitAsync(token);
            try
            {
                var response = await _influxDbService.QueryAsync(queryString, _org, token);
                var fileName = await RetrieveLastFile();
                //var fileName = await _fileService.RetrieveParsedFile($"export-{DateTime.Now.AddDays(-1):yyyy-MM-dd}.csv", StorageHelpers.ContainerName);
                var records = _localFileService.ReadFromFile(fileName);
                var cleanedUpResponses = _cleanupService.Cleanup(response, records);
                var currentDate = DateTime.Now.ToString("yyyy-MM-dd");
                var generatedFileName = $"export-{currentDate}.csv";
                await using (var writer = new StreamWriter(generatedFileName))
                await using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
                {
                    await csv.WriteRecordsAsync(cleanedUpResponses, token);
                }

                await _fileService.UploadToAzure(StorageHelpers.ContainerName, generatedFileName, token);
                var retrieveDataResponse = new RetrieveDataResponse
                {
                    Success = true,
                    GeneratedFileName = generatedFileName,
                    StartAiProcess = true,
                    TraceParent = Activity.Current!.Id!,
                    TraceState = Activity.Current!.TraceStateString!
                };

                await _daprClient.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME,
                    NameConsts.INFLUX_FINISHED_RETRIEVE_DATA,
                    retrieveDataResponse,
                    token);
            }
            finally
            {
                _semaphoreSlim.Release();
            }


            _logger.LogInformation($"Sent that retrieve of file to azaure has been finished");
        }

        private async Task<string?> RetrieveLastFile()
        {
            // Start from yesterday and go back up to 30 days
            const int maxDaysBack = 430;
            for (int daysBack = 1; daysBack <= maxDaysBack; daysBack++)
            {
                var date = DateTime.Now.AddDays(-daysBack);
                var fileName = $"export-{date:yyyy-MM-dd}.csv";
                try
                {
                    var parsedFile = await _fileService.RetrieveParsedFile(fileName, StorageHelpers.ContainerName);
                    if (!string.IsNullOrEmpty(parsedFile) && System.IO.File.Exists(parsedFile))
                    {
                        _logger.LogInformation($"Found file: {fileName} at {parsedFile}");
                        return parsedFile;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogDebug($"File not found for date {date:yyyy-MM-dd}: {ex.Message}");
                }
            }

            _logger.LogWarning($"No export file found in the last {maxDaysBack} days.");
            return null;
        }
    }
}