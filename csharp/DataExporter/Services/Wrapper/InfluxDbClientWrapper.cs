using System.Globalization;
using Common.Models.Influx;
using InfluxDB.Client;

namespace DataExporter.Services.Wrapper;

public interface IInfluxDbClientWrapper
{
    Task<List<InfluxRecord>> GetData(string queryString, string organisation, CancellationToken token);
}
public class InfluxDbClientWrapper : IInfluxDbClientWrapper
{
    private string _url;
    private string _token;

    public InfluxDbClientWrapper(string url, string token)
    {
        this._url = url;
        this._token = token;
    }

    public async Task<List<InfluxRecord>> GetData(string queryString, string organisation, CancellationToken token)
    {
        using var client = new InfluxDBClient(url: _url, token: _token);
        var query = client.GetQueryApi();
        var data = await query.QueryAsync(queryString, organisation, token);
        var paraseddata = data.SelectMany(table =>
                    table.Records.Select(record =>
                     new InfluxRecord
                     {
                         Time = DateTime.Parse(record!.GetTime()?.ToString()!, new CultureInfo("nl-BE")),
                         Watt = string.IsNullOrEmpty(record.GetValueByKey("W_value")?.ToString()) ? 0 : float.Parse(record.GetValueByKey("W_value").ToString()!),
                         Pressure = string.IsNullOrEmpty(record.GetValueByKey("state_pressure")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_pressure").ToString()!),
                         Humidity = string.IsNullOrEmpty(record.GetValueByKey("state_humidity")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("state_humidity").ToString()!),
                         Temperature = string.IsNullOrEmpty(record.GetValueByKey("°C_value")?.ToString()) ? 0 : double.Parse(record.GetValueByKey("°C_value").ToString()!),

                     })).ToList();

        return paraseddata;
    }
}
