using System.Diagnostics;
using System.Globalization;
using System.IO.Abstractions;
using Common.Exceptions;
using Common.Factory;
using Common.Helpers;
using Common.Models.AI;
using Common.Models.Responses;
using Common.Services;
using CsvHelper;
using Dapr;
using DataExporter.Services;
using Microsoft.AspNetCore.Mvc;

namespace DataExporter.Controllers
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
        private readonly IDaprClientWrapper _daprClient;
        private readonly ICleanupService _cleanupService;
        private readonly ILocalFileService _localFileService;
        private readonly SemaphoreSlim _semaphoreSlim = new(1, 1);

        [FromServices]
        public IFileSystem FileSystem { get; set; } = default!;

        public InfluxController(
           IInfluxDbService influxDbService,
           IFileService fileService,
           IConfiguration configuration,
           ICleanupService cleanupService,
           IDaprClientWrapper daprClient,
           ILocalFileService localFileService,
           ILogger<InfluxController> logger)
        {
            this._influxDbService = influxDbService;
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
            _logger.LogDebug("Start export data for date {StartDate}", strDateTime);
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
            if (string.IsNullOrEmpty(fileName)) throw new InvalidFilenameException("No file found to read from");
            _logger.LogDebug("Reading from file {FileName}", fileName);
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
            const string logPrefix = "[InfluxController:RetrieveData]";
            _logger.LogInformation("{LogPrefix} Trigger received to retrieve data from influx", logPrefix);
            _logger.LogInformation("{LogPrefix} Current activity id: {ActivityId}", logPrefix, System.Diagnostics.Activity.Current?.Id);
            _logger.LogInformation("{LogPrefix} Current activity parent id: {ParentId}",logPrefix, Activity.Current?.ParentId);
            _logger.LogInformation("{LogPrefix} Received event with traceParent {TraceParent}", logPrefix, evt.TraceParent);

            // Create a new Activity and set parent if provided
            using var activity = new Activity("InfluxController.RetrieveData");
            if (!string.IsNullOrEmpty(evt.TraceParent))
            {
                _logger.LogInformation("{LogPrefix} Setting parent id for new activity to {TraceParent}", logPrefix, evt.TraceParent);
                activity.SetParentId(evt.TraceParent);
            }
            activity.Start();
            Activity.Current = activity;

            await _semaphoreSlim.WaitAsync(token);
            try
            {
                var response = await _influxDbService.QueryAsync(queryString, _org, token);
                if (response == null || !response.Any())
                {
                    _logger.LogWarning("{LogPrefix} No data retrieved from InfluxDB", logPrefix);
                    throw new NoDataRetrievedException("No data retrieved from InfluxDB");
                }
                _logger.LogInformation("{LogPrefix} Retrieved {Count} records from InfluxDB", logPrefix, response.Count);
                var fileName = await RetrieveLastFile();
                if (string.IsNullOrEmpty(fileName)) throw new InvalidFilenameException("No file found to read from");
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
                    TraceParent = activity.Id!,
                    TraceState = activity.TraceStateString!
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
            const string logPrefix = "[InfluxController:RetrieveLastFile]";
            // Start from yesterday and go back up to 30 days
            const int maxDaysBack = 430;
            for (int daysBack = 1; daysBack <= maxDaysBack; daysBack++)
            {
                var date = DateTime.Now.AddDays(-daysBack);
                var fileName = $"export-{date:yyyy-MM-dd}.csv";
                try
                {
                    var parsedFile = await _fileService.RetrieveParsedFile(fileName, StorageHelpers.ContainerName);
                    if (!string.IsNullOrEmpty(parsedFile) && FileSystem.File.Exists(parsedFile))
                    {
                        _logger.LogInformation("{LogPrefix} Found file: {FileName} at {ParsedFile}", logPrefix, fileName, parsedFile);
                        return parsedFile;
                    }
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "{LogPrefix} File not found for date {DateTime:yyyy-MM-dd}: {ExMessage}", logPrefix, date, ex.Message);
                }
            }

            _logger.LogWarning("{LogPrefix} No export file found in the last {MaxDaysBack} days.", logPrefix, maxDaysBack);
            return null;
        }
    }
}