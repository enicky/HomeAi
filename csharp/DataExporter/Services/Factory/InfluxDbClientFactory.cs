using DataExporter.Services.Wrapper;
using InfluxDB.Client;

namespace DataExporter.Services.Factory;

public class InfluxDbClientFactory : IInfluxDbClientFactory
{
    public IInfluxDbClientWrapper CreateWrapper(string url, string token)
    {
        return new InfluxDbClientWrapper(url, token);
    }
}
