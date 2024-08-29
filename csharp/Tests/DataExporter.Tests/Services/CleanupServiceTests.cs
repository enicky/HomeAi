using Common.Models.Influx;
using DataExporter.Services;
using DataExporter.Tests.ControllerTests;
using Meziantou.Extensions.Logging.Xunit;
using Xunit.Abstractions;

namespace DataExporter.Tests.Services;

public class CleanupServiceTests : IClassFixture<TestSetup>
{
    private readonly TestSetup _testSetup;
    private readonly ITestOutputHelper _output;

    public CleanupServiceTests(TestSetup testSetup, ITestOutputHelper testOutputHelper){
        _testSetup = testSetup;
        _output = testOutputHelper;
    }

    [Fact]
    public void WhenUsingInvalidHumidity_AndNoValidFound_ThrowsException(){
        var _logger = XUnitLogger.CreateLogger<CleanupService>(_output);

        var service = new CleanupService(_logger);
        var dataToClean = new List<InfluxRecord>() { new InfluxRecord() { Humidity = 0, Pressure = 1, Temperature = 1, Time = DateTime.Now , Watt = 1}};

        Assert.Throws<InvalidDataException> (() => service.Cleanup(dataToClean));

    }


    [Fact]
    public void WhenUsingInvalidPressure_AndNoValidFound_ThrowsException(){
        var _logger = XUnitLogger.CreateLogger<CleanupService>(_output);

        var service = new CleanupService(_logger);
        var dataToClean = new List<InfluxRecord>() { new InfluxRecord() { Humidity = 1, Pressure = 0, Temperature = 1, Time = DateTime.Now , Watt = 1}};

        Assert.Throws<InvalidDataException> (() => service.Cleanup(dataToClean));

    }

     [Fact]
    public void WhenUsingInvalidTemperature_AndNoValidFound_ThrowsException(){
        var _logger = XUnitLogger.CreateLogger<CleanupService>(_output);

        var service = new CleanupService(_logger);
        var dataToClean = new List<InfluxRecord>() { new InfluxRecord() { Humidity = 1, Pressure = 1, Temperature = 0, Time = DateTime.Now , Watt = 1}};

        Assert.Throws<InvalidDataException> (() => service.Cleanup(dataToClean));

    }

    [Fact]
    public void WhenUsingInvalidWatt_AndNoValidFound_ThrowsException(){
        var _logger = XUnitLogger.CreateLogger<CleanupService>(_output);

        var service = new CleanupService(_logger);
        var dataToClean = new List<InfluxRecord>() { new InfluxRecord() { Humidity = 1, Pressure = 1, Temperature = 1, Time = DateTime.Now , Watt = 0}};

        Assert.Throws<InvalidDataException> (() => service.Cleanup(dataToClean));

    }
}
