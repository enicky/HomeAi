using System.IO.Abstractions.TestingHelpers;
using System.Security.Policy;
using Common.Models.Influx;
using DataExporter.Services;
using DataExporter.Tests.ControllerTests;
using Newtonsoft.Json;
using Xunit.Abstractions;

namespace DataExporter.Tests.File;

public class LocalFileServiceTests : IClassFixture<TestSetup>
{
    private readonly TestSetup _testSetup;
    private readonly ITestOutputHelper _output;
    private readonly MockFileSystem _mockFileSystem = new();
    public LocalFileServiceTests(TestSetup testSetup, ITestOutputHelper testOutputHelper)
    {
        _testSetup = testSetup;
        _output = testOutputHelper;
        _output.WriteLine("Start testing of local file services");

    }
    [Fact]
    public async Task LocalFileService_Should_Return()
    {
        var sut = CreateSut();
        var dataToWrite = new List<InfluxRecord> { new InfluxRecord() { Humidity = 1, Pressure = 1, Temperature = 1, Watt = 2, Time = DateTime.Today } };
        _output.WriteLine($"Start testing write to file with data {JsonConvert.SerializeObject(dataToWrite)}");
        await sut.WriteToFile("test.txt", dataToWrite, default);
        _output.WriteLine("Finished testing writing to file");
        var file = _mockFileSystem.GetFile("test.txt");
        var content = System.Text.Encoding.Default.GetString(file.Contents);
        _output.WriteLine($"content : {content}");
        var mustBe = $@"Time,Watt,Humidity,Pressure,Temperature{NewLine()}{DateTime.Today.ToString("MM/dd/yyyy HH:mm:ss")},2,1,1,1{NewLine()}";
        Assert.Equal(mustBe, content);
    }

    [Fact]
    public async Task IfSourceFiles_DoesNotExist_WeDontStartCopying()
    {
        var sut = CreateSut();
        await sut.CopyFile("a", "b");
        Assert.False(_mockFileSystem.FileExists("b"));
    }
    [Fact]
    public async Task IfSourceFileDoesExist_WeDoSomethingElse()
    {
        var content = $"abc";

        var fileSystem = new MockFileSystem(new Dictionary<string, MockFileData>{
            {"test.txt", new MockFileData(content)}
        });
        var sut = new LocalFileService(fileSystem);
        await sut.CopyFile("test.txt", "test2.txt");
        Assert.True(_mockFileSystem.File.Exists("test2.txt"));
    }

    [Fact]
    public async Task IfTargetFileExists_DeleteFile()
    {
        var content = $"abc";

        var fileSystem = new MockFileSystem(new Dictionary<string, MockFileData>{
            {"test.txt", new MockFileData(content)},
            {"target.txt", new MockFileData(content)}
        });
        var sut = new LocalFileService(fileSystem);
        await sut.CopyFile("test.txt", "target.txt");
        Assert.True(_mockFileSystem.File.Exists("target.txt"));
    }   

    [Fact]
    public void LocalFileService_ReadFromFile_ShouldWork()
    {

        var content = $"Time,Watt,Humidity,Pressure,Temperature{Environment.NewLine}08/18/2024 02:01:00,390.85455,93,1010.1,21";

        var fileSystem = new MockFileSystem(new Dictionary<string, MockFileData>{
            {"test.txt", new MockFileData(content)}
        });
        var sut = new LocalFileService(fileSystem);
        var x = sut.ReadFromFile("test.txt");
        Assert.NotNull(x);

    }

    private static string NewLine()
    {
        return $"{(char)0x0D}{(char)0x0A}";
    }

    private LocalFileService CreateSut()
    {
        var sut = new LocalFileService(_mockFileSystem);
        return sut;
    }
}
