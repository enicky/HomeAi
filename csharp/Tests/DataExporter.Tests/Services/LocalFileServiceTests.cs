using System.IO.Abstractions.TestingHelpers;
using Common.Models.Influx;
using DataExporter.Services;
using DataExporter.Tests.ControllerTests;
using Newtonsoft.Json;
using Xunit.Abstractions;

namespace DataExporter.Tests.File;

public class LocalFileServiceTests: IClassFixture<TestSetup>
{
    private readonly TestSetup _testSetup;
    private readonly ITestOutputHelper _output;
    private readonly MockFileSystem _mockFileSystem =new();
    public LocalFileServiceTests(TestSetup testSetup, ITestOutputHelper testOutputHelper){
        _testSetup = testSetup;
        _output  = testOutputHelper;
        _output.WriteLine("Start testing of local file services");
        
    }
    [Fact]
    public void LocalFileService_Should_Return(){
        var sut = CreateSut();
        var dataToWrite = new List<InfluxRecord> { new InfluxRecord() { Humidity = 1, Pressure = 1, Temperature = 1, Watt = 2, Time = DateTime.Today}};
        _output.WriteLine($"Start testing write to file with data {JsonConvert.SerializeObject(dataToWrite)}");
        sut.WriteToFile("test.txt", dataToWrite, default);
        _output.WriteLine("Finished testing writing to file");
        var file = _mockFileSystem.GetFile("test.txt");
        var content = System.Text.Encoding.Default.GetString(file.Contents);
        _output.WriteLine($"content : {content}");
        var mustBe = $@"Time,Watt,Humidity,Pressure,Temperature{NewLine()}{DateTime.Today.ToString("MM/dd/yyyy HH:mm:ss")},2,1,1,1{NewLine()}";
        Assert.Equal(mustBe, content);
    }

    private static string NewLine()
    {
        return $"{(char)0x0D}{(char)0x0A}";
    }

    private ILocalFileService CreateSut(){
        var sut = new LocalFileService(_mockFileSystem); 
        return sut;
    }
}