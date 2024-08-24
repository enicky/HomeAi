
using System.Globalization;
using System.IO.Abstractions;
using Common.Models.Influx;
using CsvHelper;

namespace DataExporter.Services;

public interface ILocalFileService
{
    Task WriteToFile(string generatedFileName, List<InfluxRecord>? cleanedUpResponses, CancellationToken token);
}
public class LocalFileService : ILocalFileService
{
    private readonly IFileSystem _fileSystem;

    public LocalFileService() : this(new FileSystem()){}
    public LocalFileService(IFileSystem fileSystem){
        _fileSystem = fileSystem;
    }
    public async Task WriteToFile(string generatedFileName, List<InfluxRecord>? cleanedUpResponses, CancellationToken token)
    {
        
        using var writer = _fileSystem.File.CreateText(generatedFileName);// new StreamWriter(generatedFileName);
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
        await csv.WriteRecordsAsync(cleanedUpResponses, token);
    }
}
