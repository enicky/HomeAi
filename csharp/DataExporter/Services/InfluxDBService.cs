using Common.Models.Influx;
using InfluxDB.Client;


namespace app.Services
{
    public interface IInfluxDbService
    {
        void Write(Action<WriteApi> action);
        Task<List<InfluxRecord>> QueryAsync(string query, string organisation);
    }

    public class InfluxDBService : IInfluxDbService
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

        public async Task<List<InfluxRecord>> QueryAsync(string queryString, string organisation)
        {
            using var client = new InfluxDBClient(url: _url, token: _token);
            var query = client.GetQueryApi();
            var data = await query.QueryAsync(queryString, organisation);
            var paraseddata = data.SelectMany(table =>
                        table.Records.Select(record =>
                         new InfluxRecord
                         {
                             Time = DateTime.Parse(record!.GetTime()?.ToString()!),
                             Watt = string.IsNullOrEmpty(record.GetValueByKey("W_value")?.ToString()) ? 0 : float.Parse(record.GetValueByKey("W_value").ToString()!),
                             Pressure = string.IsNullOrEmpty(record.GetValueByKey("state_pressure")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_pressure").ToString()!),
                             Humidity = string.IsNullOrEmpty(record.GetValueByKey("state_humidity")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_humidity").ToString()!),
                             Temperature = string.IsNullOrEmpty(record.GetValueByKey("°C_value")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("°C_value").ToString()!),

                         })).ToList();

            return paraseddata;
            //return await action(query);
        }
    }
}