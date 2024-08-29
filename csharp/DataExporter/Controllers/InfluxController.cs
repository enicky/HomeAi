using System.Globalization;
using app.Services;
using Common.Helpers;
using Common.Models;
using Common.Models.Influx;
using Common.Models.Responses;
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

        private readonly IInfluxDbService influxDBService;
        private readonly string _org;
        private readonly IFileService _fileService;
        private readonly ILogger<InfluxController> _logger;
        private readonly DaprClient _daprClient;
        private readonly ICleanupService _cleanupService;
        private readonly ILocalFileService _localFileService;

        public InfluxController(IInfluxDbService influxDBService, 
                    IFileService fileService, 
                    IConfiguration configuration,
                    ICleanupService cleanupService,
                    DaprClient daprClient,
                    ILocalFileService localFileService,
                    ILogger<InfluxController> logger)
        {
            this.influxDBService = influxDBService;
            _org = configuration.GetValue<string>("InfluxDB:Org")!;
            _fileService = fileService;
            _logger = logger;
            _daprClient = daprClient;
            _cleanupService = cleanupService;
            _localFileService = localFileService;

        }

        [HttpGet("ExportDataForDate")]
        public async Task<IActionResult> ExportDataForDate(DateTime startDate, CancellationToken token){
            var strDateTime = startDate.ToStartDayString();
            _logger.LogDebug("Start export data for date {startDate}", strDateTime);
            var strStartDate = startDate.AddDays(-1).ToStartDayString();
            var strTomorrowStartDate = startDate.ToStartDayString();

            var q = "import \"experimental\"" +
            " from(bucket: \"home_assistant\")" +
            " |> range(start: "+strStartDate+", stop: "+strTomorrowStartDate+")" +
            " |> filter(fn: (r) => r[\"entity_id\"] == \"forecast_home_2\" or r[\"entity_id\"] == \"warmtepomp_power\" or r[\"entity_id\"] == \"smoke_detector_device_17_temperature\")" +
            " |> filter(fn: (r) =>  r[\"_field\"] == \"humidity\" or r[\"_field\"] == \"pressure\" or r[\"_field\"] == \"value\")" +
            " |> aggregateWindow(every: 1m, fn: mean, createEmpty: true)" +
            " |> fill(usePrevious: true)" +
            " |> keep(columns: [\"_field\", \"_measurement\", \"_value\", \"_time\"])" +
            " |> pivot(" +
            "     rowKey: [\"_time\"], columnKey: [\"_measurement\", \"_field\"], valueColumn: \"_value\")" +

            " |> yield(name: \"values\")";

            var response = await influxDBService.QueryAsync(q, _org, token);
            var cleanedUpResponses = _cleanupService.Cleanup(response.ToList());
            var currentDate = startDate.ToString("yyyy-MM-dd");
            var generatedFileName = $"export-{currentDate}.csv";
            await _localFileService.WriteToFile(generatedFileName, cleanedUpResponses,token );
            await _fileService.UploadToAzure(StorageHelpers.ContainerName, generatedFileName, token);
           
            _logger.LogDebug($"Finished uploading to Azure");
            return Ok();

        }

        [Topic(pubsubName: NameConsts.INFLUX_PUBSUB_NAME, name: NameConsts.INFLUX_RETRIEVE_DATA)]
        [HttpPost(NameConsts.INFLUX_RETRIEVE_DATA)]
        public async Task RetrieveData(CancellationToken token)
        {
            _logger.LogDebug("Trigger received to retrieve data from influx");

            var response = await influxDBService.QueryAsync(queryString, _org, token);

           
            var cleanedUpResponses = _cleanupService.Cleanup(response);
            var currentDate = DateTime.Now.ToString("yyyy-MM-dd");
            var generatedFileName = $"export-{currentDate}.csv";
            using (var writer = new StreamWriter(generatedFileName))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                await csv.WriteRecordsAsync(cleanedUpResponses, token);
            }
            await _fileService.UploadToAzure(StorageHelpers.ContainerName, generatedFileName, token);
            
            var retrieveDataResponse = new RetrieveDataResponse
            {
                Success = true,
                GeneratedFileName = generatedFileName,
                StartAiProcess = true

            };  
            
            await _daprClient.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME, 
                                NameConsts.INFLUX_FINISHED_RETRIEVE_DATA, 
                                retrieveDataResponse,
                                token);

            _logger.LogDebug($"Sent that retrieve of file to azaure has been finished");
        }
    }
}