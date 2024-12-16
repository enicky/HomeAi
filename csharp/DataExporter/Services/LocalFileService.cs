
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.IO.Abstractions;
using Common.Models.Influx;
using CsvHelper;

namespace DataExporter.Services;

public interface ILocalFileService
{
    Task WriteToFile(string generatedFileName, List<InfluxRecord>? cleanedUpResponses, CancellationToken token);
    List<InfluxRecord> ReadFromFile(string fileName);
    Task CopyFile(string source, string target);
}
public class LocalFileService : ILocalFileService
{
    private readonly IFileSystem _fileSystem;

    [ExcludeFromCodeCoverage]
    public LocalFileService() : this(new FileSystem()) { }
    public LocalFileService(IFileSystem fileSystem)
    {
        _fileSystem = fileSystem;
    }

    public Task CopyFile(string source, string target)
    {
        if (_fileSystem.File.Exists(target))
        {
            _fileSystem.File.Delete(target);
        }
        if (_fileSystem.File.Exists(source))
        {
            _fileSystem.File.Copy(source, target,true);
        }
        return Task.CompletedTask;
    }
    public async Task WriteToFile(string generatedFileName, List<InfluxRecord>? cleanedUpResponses, CancellationToken token)
    {
        using var writer = _fileSystem.File.CreateText(generatedFileName);
        using var csv = new CsvWriter(writer, CultureInfo.InvariantCulture);
        await csv.WriteRecordsAsync(cleanedUpResponses, token);
    }

    public List<InfluxRecord> ReadFromFile(string fileName)
    {
        using var reader = _fileSystem.File.OpenText(fileName);
        using var csvreader = new CsvReader(reader, CultureInfo.InvariantCulture);
        var records = csvreader.GetRecords<InfluxRecord>();
        return records.ToList();
    }
}
