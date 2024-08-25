using DataExporter.Services.Wrapper;
using InfluxDB.Client;

namespace DataExporter.Services.Factory;

public interface IInfluxDbClientFactory
{
    IInfluxDbClientWrapper CreateWrapper(string url, string token);
}
