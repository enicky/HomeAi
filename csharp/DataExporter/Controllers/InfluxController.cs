using System.Globalization;
using app.Services;
using Common.Helpers;
using Common.Models.Influx;
using Common.Models.Responses;
using CsvHelper;
using Dapr;
using Dapr.Client;
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

        public InfluxController(InfluxDBService influxDBService, IFileService fileService, IConfiguration configuration, ILogger<InfluxController> logger)
        {
            this.influxDBService = influxDBService;
            _org = configuration.GetValue<string>("InfluxDB:Org")!;
            _fileService = fileService;
            _logger = logger;
            
        }
    

        [Topic(pubsubName:NameConsts.INFLUX_PUBSUB_NAME, name:NameConsts.INFLUX_RETRIEVE_DATA)]
        [HttpGet("retrievedata")]
        public async Task RetrieveData(CancellationToken token)
        {
            var response = await influxDBService.QueryAsync(async query =>{
                var data = await query.QueryAsync(queryString, _org);
                return data.SelectMany(table => 
                        table.Records.Select( record =>
                         new InfluxRecord{
                            Time = DateTime.Parse(record!.GetTime()?.ToString()!),
                            Watt = string.IsNullOrEmpty(record.GetValueByKey("W_value")?.ToString()) ? 0 : float.Parse(record.GetValueByKey("W_value").ToString()!),
                            Pressure = string.IsNullOrEmpty(record.GetValueByKey("state_pressure")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_pressure").ToString()!),
                            Humidity = string.IsNullOrEmpty(record.GetValueByKey("state_humidity")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_humidity").ToString()!),
                            Temperature = string.IsNullOrEmpty(record.GetValueByKey("°C_value")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("°C_value").ToString()!),
                            
                         }));
            });
            var currentDate = DateTime.Now.ToString("yyyy-MM-dd");
            var generatedFileName = $"export-{currentDate}.csv";
            using (var writer = new StreamWriter(generatedFileName))
            using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
            {
                csv.WriteRecords(response);
            }
            _logger.LogDebug("Ensuring container exists {containerName}", StorageHelpers.ContainerName);
            var result = await _fileService.EnsureContainer(StorageHelpers.ContainerName) ?? throw new Exception("result is null");
            _logger.LogDebug("Result of ensureContainer : {result}. Start uploading to Azure", result);
            await _fileService.UploadFromFileAsync(result, generatedFileName);
            _logger.LogDebug($"Finished uploading to Azure");
            
            var retrieveDataResponse = new RetrieveDataResponse
            {
                Success = true,
                Value = response,
                GeneratedFileName = generatedFileName
            };

            using var client = new DaprClientBuilder().Build();
            await client.PublishEventAsync(NameConsts.INFLUX_PUBSUB_NAME,
                            NameConsts.INFLUX_FINISHED_RETRIEVE_DATA,
                            retrieveDataResponse, 
                            token);

            _logger.LogDebug($"Sent that retrieve of file to azaure has been finished");

            
        }
    }
}