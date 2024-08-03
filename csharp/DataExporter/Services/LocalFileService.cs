
using System.Globalization;
using Common.Models.Influx;
using CsvHelper;

namespace DataExporter.Services;

public interface ILocalFileService
{
    Task WriteToFile(string generatedFileName, List<InfluxRecord>? cleanedUpResponses, CancellationToken token);
}
public class LocalFileService : ILocalFileService
{
    public async Task WriteToFile(string generatedFileName, List<InfluxRecord>? cleanedUpResponses, CancellationToken token)
    {
        using var writer = new StreamWriter(generatedFileName);
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
        await csv.WriteRecordsAsync(cleanedUpResponses, token);
    }
}
