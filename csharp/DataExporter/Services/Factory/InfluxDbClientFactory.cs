using DataExporter.Services.Wrapper;
using InfluxDB.Client;

namespace DataExporter.Services.Factory;

public class InfluxDbClientFactory : IInfluxDbClientFactory
{
    public InfluxDBClient CreateClient(string url, string token)
    {
        return new InfluxDBClient(url:url, token: token);
    }

    public IInfluxDbClientWrapper CreateWrapper(string url, string token)
    {
        return new InfluxDbClientWrapper(url, token);
    }
}
