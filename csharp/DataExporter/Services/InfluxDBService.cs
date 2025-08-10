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
            logger.LogInformation($"start query");
            var wrapper = factory.CreateWrapper(_url, _token);

            var x = await wrapper.GetData(queryString, organisation, token);        
            logger.LogInformation("Get data result : {DataResult}", x);
            return x;    
        }
    }
}