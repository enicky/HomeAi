using Common.Models.Influx;
using DataExporter.Services;
using Microsoft.Extensions.Logging;
using Moq;

namespace DataExporter.Tests;

public class CleanupServiceTests
{
    private readonly ILogger<CleanupService> _logger;

    public CleanupServiceTests(){
        _logger = Mock.Of<ILogger<CleanupService>>();
    }

    [Fact]
    public void WhenHaving2Records_And1IsIncorrect_ItCleansData()
    {
        var sut = CreateSut();

        var setup = new List<InfluxRecord>(){
            new InfluxRecord(){
                Humidity = 0, Pressure = 0, Temperature = 0, Watt = 0, Time = System.DateTime.Now
            },
            new InfluxRecord(){
                Humidity = 10, Pressure = 10, Temperature = 10, Watt = 10, Time = System.DateTime.Now.AddHours(1)
            },
        };
        var result = sut.Cleanup(setup);
        Assert.NotNull(result);
        Assert.NotEmpty(result);
        Assert.Equal(2, result.Count());
        Assert.Equal(10, result.First().Humidity);
        Assert.Equal(10, result.First().Pressure);
        Assert.Equal(10, result.First().Temperature);
        Assert.Equal(10, result.First().Watt);
    }

    private ICleanupService CreateSut()
    {
        var sut = new CleanupService(_logger);

        return sut;
    }
}