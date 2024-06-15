using InfluxDB.Client;


namespace app.Services
{
    public class InfluxDBService
    {
        private readonly string _token;
        private readonly string _url;

        public InfluxDBService(IConfiguration configuration)
        {
            _token = configuration.GetValue<string>("InfluxDB_TOKEN")!;
            _url = configuration.GetValue<string>("InfluxDB:Url")!;
        }

        public void Write(Action<WriteApi> action)
        {
            using var client = new InfluxDBClient(url: _url, token: _token);
            using var write = client.GetWriteApi();
            action(write);
        }

        public async Task<T> QueryAsync<T>(Func<QueryApi, Task<T>> action)
        {
            using var client = new InfluxDBClient(url: _url, token: _token);
            var query = client.GetQueryApi();
            return await action(query);
        }
    }
}