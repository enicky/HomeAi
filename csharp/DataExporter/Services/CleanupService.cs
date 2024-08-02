using Common.Models.Influx;
using Microsoft.Identity.Client;

namespace DataExporter.Services;


public interface ICleanupService
{
    List<InfluxRecord> Cleanup(List<InfluxRecord> lst);
}
public class CleanupService : ICleanupService
{
    private readonly ILogger<CleanupService> _logger;
    private readonly double epsilon = 0.001;

    public CleanupService(ILogger<CleanupService> logger)
    {
        _logger = logger;

    }
    public List<InfluxRecord> Cleanup(List<InfluxRecord> lst)
    {
        var cleaned = lst.Select(x =>
        {
            x.Humidity = GetValidHumidity(x.Humidity, lst);
            x.Pressure = GetValidPressure(x.Pressure, lst);
            x.Temperature = GetValidTemperature(x.Temperature, lst);
            x.Watt = GetValidWatt(x.Watt, lst);
            return x;
        });
        return cleaned.ToList();
    }

    private float GetValidWatt(float watt, List<InfluxRecord> data)
    {
        
        if(watt > 0) return watt;
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs( q.Watt) >= epsilon);
        if (firstValidValue == null)
        {
            _logger.LogError("No valid Watt found for today!!");
            throw new InvalidDataException("No valid Watt found for today!");
        }
        var floatValue = firstValidValue.Watt;
        return floatValue;
    }

    private double GetValidTemperature(double temperature, List<InfluxRecord> data)
    {
        if(temperature > 0) return temperature;
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs(q.Temperature) >= epsilon);
        if (firstValidValue == null)
        {
            _logger.LogError("No valid temperature found for today!!");
            throw new InvalidDataException("No valid temperature found for today!");
        }
        var floatValue = firstValidValue.Temperature;
        return floatValue;
    }

    private double GetValidPressure(double pressure, List<InfluxRecord> data)
    {
        if(pressure > 100) return pressure;
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs(q.Pressure) >= epsilon);
        //_logger.LogInformation("Found valid pressure : {pressure}", firstValidValue.Pressure);
        if (firstValidValue == null)
        {
            _logger.LogError("No valid pressure found for today!!");
            throw new InvalidDataException("No valid pressure found for today!");
        }
        var floatValue = firstValidValue.Pressure;
        return floatValue;
    }

    private double GetValidHumidity(double humidity, List<InfluxRecord> data)
    {
        if(humidity > 5) return humidity;
        // hum was null or empty 
        // get the first valid value from the list
        //_logger.LogInformation("No Valid humidity found. Search for the first(next) valid one.");
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs(q.Humidity) >= epsilon);
        //_logger.LogInformation("Found valid humidity : {humidity}", firstValidValue.Humidity);
        if (firstValidValue == null)
        {
            _logger.LogError($"No valid humidity found for today ... is this correct ??");
            throw new InvalidDataException("No valid Humidity found for today");
        }
        var floatValue = firstValidValue.Humidity;
        return floatValue;
    }
}
