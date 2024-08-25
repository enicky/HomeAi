using DataExporter.Services.Wrapper;
using InfluxDB.Client;

namespace DataExporter.Services.Factory;

public interface IInfluxDbClientFactory
{
    InfluxDBClient CreateClient(string url, string token);
    IInfluxDbClientWrapper CreateWrapper(string url, string token);
}
