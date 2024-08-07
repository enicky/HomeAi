using System.Globalization;
using System.Net.Mime;
using System.Text.Json;
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
        private string queryString = "import \"experimental\"" +
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

        private readonly InfluxDBService influxDBService;
        private readonly string _org;
        private readonly IFileService _fileService;
        private readonly ILogger<InfluxController> _logger;
        private readonly DaprClient _daprClient;
        private readonly ICleanupService _cleanupService;

        public InfluxController(InfluxDBService influxDBService, 
                    IFileService fileService, 
                    IConfiguration configuration,
                    ICleanupService cleanupService,
                    DaprClient daprClient,
                    ILogger<InfluxController> logger)
        {
            this.influxDBService = influxDBService;
            _org = configuration.GetValue<string>("InfluxDB:Org")!;
            _fileService = fileService;
            _logger = logger;
            _daprClient = daprClient;
            _cleanupService = cleanupService;

        }

        [HttpGet("ExportDataForDate")]
        public async Task ExportDataForDate(DateTime startDate, CancellationToken token){
            _logger.LogInformation($"Start export data for date {startDate}");
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
            _logger.LogInformation($"Using the following query : {q}");
            var response = await influxDBService.QueryAsync(async query => {
                var data = await query.QueryAsync(q, _org);
                return data.SelectMany(table =>
                        table.Records.Select(record =>
                         new InfluxRecord
                         {
                             Time = DateTime.Parse(record!.GetTime()?.ToString()!),
                             Watt = string.IsNullOrEmpty(record.GetValueByKey("W_value")?.ToString()) ? 0 : float.Parse(record.GetValueByKey("W_value").ToString()!),
                             Pressure = string.IsNullOrEmpty(record.GetValueByKey("state_pressure")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_pressure").ToString()!),
                             Humidity = string.IsNullOrEmpty(record.GetValueByKey("state_humidity")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_humidity").ToString()!),
                             Temperature = string.IsNullOrEmpty(record.GetValueByKey("°C_value")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("°C_value").ToString()!),

                         }));
            });
            var cleanedUpResponses = _cleanupService.Cleanup(response.ToList());
            var currentDate = startDate.ToString("yyyy-MM-dd");
            var generatedFileName = $"export-{currentDate}.csv";
            _logger.LogInformation($"Start writing file to {generatedFileName}");
            using (var writer = new StreamWriter(generatedFileName))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                await csv.WriteRecordsAsync(cleanedUpResponses);
            }
            _logger.LogInformation("Ensuring container exists {containerName}", StorageHelpers.ContainerName);
            var result = await _fileService.EnsureContainer(StorageHelpers.ContainerName) ?? throw new Exception("result is null");
            _logger.LogInformation("Result of ensureContainer : {result}. Start uploading to Azure", result);
            await _fileService.UploadFromFileAsync(result, generatedFileName);
            _logger.LogInformation($"Finished uploading to Azure");

        }

        [Dapr.Topic(NameConsts.INFLUX_PUBSUB_NAME, "test" )]
        [HttpPost("test")]
        public async Task<IActionResult> Test([FromBody] Order o){
            if(o is not null){
                _logger.LogInformation($"Reeived order {o.Id} -> {o.Title}");
                await _daprClient.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME, "testreply", new RetrieveDataResponse{Success=true, GeneratedFileName="test.csv", StartAiProcess=false});
                _logger.LogInformation("Replied success to topic testreply");
                return Ok();
            }
            return BadRequest();
        }


        [Topic(pubsubName: NameConsts.INFLUX_PUBSUB_NAME, name: NameConsts.INFLUX_RETRIEVE_DATA)]
        [HttpPost(NameConsts.INFLUX_RETRIEVE_DATA)]
        public async Task RetrieveData()
        {
            _logger.LogInformation("Trigger received to retrieve data from influx");
            var response = await influxDBService.QueryAsync(async query =>
            {
                var data = await query.QueryAsync(queryString, _org);
                return data.SelectMany(table =>
                        table.Records.Select(record =>
                         new InfluxRecord
                         {
                             Time = DateTime.Parse(record!.GetTime()?.ToString()!),
                             Watt = string.IsNullOrEmpty(record.GetValueByKey("W_value")?.ToString()) ? 0 : float.Parse(record.GetValueByKey("W_value").ToString()!),
                             Pressure = string.IsNullOrEmpty(record.GetValueByKey("state_pressure")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_pressure").ToString()!),
                             Humidity = string.IsNullOrEmpty(record.GetValueByKey("state_humidity")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_humidity").ToString()!),
                             Temperature = string.IsNullOrEmpty(record.GetValueByKey("°C_value")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("°C_value").ToString()!),

                         }));
            });
            var cleanedUpResponses = _cleanupService.Cleanup(response.ToList());
            var currentDate = DateTime.Now.ToString("yyyy-MM-dd");
            var generatedFileName = $"export-{currentDate}.csv";
            _logger.LogInformation($"Start writing file to {generatedFileName}");
            using (var writer = new StreamWriter(generatedFileName))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                await csv.WriteRecordsAsync(cleanedUpResponses);
            }
            _logger.LogInformation("Ensuring container exists {containerName}", StorageHelpers.ContainerName);
            var blobContainerClient = await _fileService.EnsureContainer(StorageHelpers.ContainerName) ?? throw new Exception("result is null");
            _logger.LogInformation("Result of ensureContainer : {result}. Start uploading to Azure", blobContainerClient);
            await _fileService.UploadFromFileAsync(blobContainerClient, generatedFileName);
            _logger.LogInformation($"Finished uploading to Azure");

            var retrieveDataResponse = new RetrieveDataResponse
            {
                Success = true,
                GeneratedFileName = generatedFileName,
                StartAiProcess = true

            };
            var metaData = new Dictionary<string, string>(){
                { "test", "value"},
            };
            
            await _daprClient.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME, 
                                NameConsts.INFLUX_FINISHED_RETRIEVE_DATA, 
                                retrieveDataResponse);

            _logger.LogInformation($"Sent that retrieve of file to azaure has been finished");
        }
    }
}