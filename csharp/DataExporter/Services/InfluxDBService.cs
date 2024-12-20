using System.Globalization;
using Common.Models.Influx;
using DataExporter.Services.Factory;
using InfluxDB.Client;


namespace app.Services
{
    public interface IInfluxDbService
    {
        Task<List<InfluxRecord>> QueryAsync(string queryString, string organisation, CancellationToken token);
    }

    public class InfluxDBService : IInfluxDbService
    {
        private readonly string _token;
        private readonly string _url;
        private readonly IInfluxDbClientFactory _factory;
        private readonly ILogger<InfluxDBService> _logger;

        public InfluxDBService(IConfiguration configuration, IInfluxDbClientFactory factory, ILogger<InfluxDBService> logger)
        {
            _logger = logger;
            _factory = factory;
            _token = configuration.GetValue<string>("InfluxDB_TOKEN")!;
            _url = configuration.GetValue<string>("InfluxDB:Url")!;
        }
        public async Task<List<InfluxRecord>> QueryAsync(string queryString, string organisation, CancellationToken token)
        {
            _logger.LogInformation($"start query");
            var wrapper = _factory.CreateWrapper(_url, _token);

            var x = await wrapper.GetData(queryString, organisation, token);        
            _logger.LogInformation("Get data result : {DataResult}", x);
            return x;    
        }
    }
}