using Common.Models.Influx;
using DataExporter.Services.Factory;

namespace DataExporter.Services
{
    public interface IInfluxDbService
    {
        Task<List<InfluxRecord>> QueryAsync(string queryString, string organisation, CancellationToken token);
    }

    public class InfluxDbService(
        IConfiguration configuration,
        IInfluxDbClientFactory factory,
        ILogger<InfluxDbService> logger)
        : IInfluxDbService
    {
        private readonly string _token = configuration.GetValue<string>("InfluxDB_TOKEN")!;
        private readonly string _url = configuration.GetValue<string>("InfluxDB:Url")!;

        public async Task<List<InfluxRecord>> QueryAsync(string queryString, string organisation, CancellationToken token)
        {
            const string logPrefix = "[InfluxDbService:QueryAsync]";
            logger.LogInformation("{LogPrefix} Start querying influxdb at url: {Url} for organisation: {Organisation}", logPrefix, _url, organisation);
            var wrapper = factory.CreateWrapper(_url, _token);

            var x = await wrapper.GetData(queryString, organisation, token);        
            logger.LogInformation("{LogPrefix} Get data result : {DataResult}", logPrefix, x);
            return x;    
        }
    }
}