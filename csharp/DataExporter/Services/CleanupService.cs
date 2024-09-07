using Common.Models.Influx;
using Microsoft.Identity.Client;

namespace DataExporter.Services;


public interface ICleanupService
{
    List<InfluxRecord> Cleanup(List<InfluxRecord> lst, List<InfluxRecord> recordsDayMinusOne);
}
public class CleanupService : ICleanupService
{
    private readonly ILogger<CleanupService> _logger;
    private readonly double epsilon = 0.001;

    public CleanupService(ILogger<CleanupService> logger)
    {
        _logger = logger;

    }
    public List<InfluxRecord> Cleanup(List<InfluxRecord> lst, List<InfluxRecord> recordsDayMinusOne)
    {
        var cleaned = lst.Select(x =>
        {
            x.Humidity = GetValidHumidity(x.Humidity, lst, recordsDayMinusOne);
            x.Pressure = GetValidPressure(x.Pressure, lst, recordsDayMinusOne);
            x.Temperature = GetValidTemperature(x.Temperature, lst, recordsDayMinusOne);
            x.Watt = GetValidWatt(x.Watt, lst, recordsDayMinusOne);
            return x;
        });
        return cleaned.ToList();
    }

    private float GetValidWatt(float watt, List<InfluxRecord> data, List<InfluxRecord>? recordsDayMinusOne)
    {

        if (watt > 0) return watt;
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs(q.Watt) >= epsilon);
        if (firstValidValue == null)
        {
            if (recordsDayMinusOne == null || recordsDayMinusOne.Count == 0)
            {
                _logger.LogError("No valid Watt found for today!!");
                throw new InvalidDataException("No valid Watt found for today!");
            }
            recordsDayMinusOne.Sort((x, y) => DateTime.Compare(y.Time, x.Time));

            firstValidValue = recordsDayMinusOne.Find(q => Math.Abs(q.Watt) >= epsilon);
            if (firstValidValue == null)
            {
                _logger.LogError($"No valid Watt found for YESTERDAY ... is this correct ??");
                throw new InvalidDataException("No valid Watt found for yesterday");
            }
        }
        var floatValue = firstValidValue.Watt;
        return floatValue;
    }

    private double GetValidTemperature(double temperature, List<InfluxRecord> data, List<InfluxRecord>? recordsDayMinusOne)
    {
        if (temperature > 0) return temperature;
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs(q.Temperature) >= epsilon);
        if (firstValidValue == null)
        {
            if (recordsDayMinusOne == null || recordsDayMinusOne.Count == 0)
            {
                _logger.LogError("No valid temperature found for today!!");
                throw new InvalidDataException("No valid temperature found for today!");
            }
            recordsDayMinusOne.Sort((x, y) => DateTime.Compare(y.Time, x.Time));

            firstValidValue = recordsDayMinusOne.Find(q => Math.Abs(q.Temperature) >= epsilon);
            if (firstValidValue == null)
            {
                _logger.LogError($"No valid Temperature found for YESTERDAY ... is this correct ??");
                throw new InvalidDataException("No valid Temperature found for yesterday");
            }
        }
        var floatValue = firstValidValue.Temperature;
        return floatValue;
    }

    private double GetValidPressure(double pressure, List<InfluxRecord> data, List<InfluxRecord>? recordsDayMinusOne)
    {
        if (pressure > 100) return pressure;
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs(q.Pressure) >= epsilon);
        if (firstValidValue == null)
        {
            if (recordsDayMinusOne == null || recordsDayMinusOne.Count == 0)
            {
                _logger.LogError("No valid pressure found for today!!");
                throw new InvalidDataException("No valid pressure found for today!");
            }
            recordsDayMinusOne.Sort((x, y) => DateTime.Compare(y.Time, x.Time));

            firstValidValue = recordsDayMinusOne.Find(q => Math.Abs(q.Pressure) >= epsilon);
            if (firstValidValue == null)
            {
                _logger.LogError($"No valid pressure found for YESTERDAY ... is this correct ??");
                throw new InvalidDataException("No valid pressure found for yesterday");
            }
        }
        var floatValue = firstValidValue.Pressure;
        return floatValue;
    }

    private double GetValidHumidity(double humidity, List<InfluxRecord> data, List<InfluxRecord>? recordsDayMinusOne)
    {
        if (humidity > 5) return humidity;
        // hum was null or empty 
        // get the first valid value from the list
        data.Sort((x, y) => DateTime.Compare(x.Time, y.Time));
        var firstValidValue = data.Find(q => Math.Abs(q.Humidity) >= epsilon);
        if (firstValidValue == null)
        {
            if (recordsDayMinusOne == null || recordsDayMinusOne.Count == 0)
            {
                _logger.LogError($"No valid humidity found for today ... is this correct ??");
                throw new InvalidDataException("No valid Humidity found for today");
            }
            recordsDayMinusOne.Sort((x, y) => DateTime.Compare(y.Time, x.Time));

            firstValidValue = recordsDayMinusOne.Find(q => Math.Abs(q.Humidity) >= epsilon);
            if (firstValidValue == null)
            {
                _logger.LogError($"No valid humidity found for YESTERDAY ... is this correct ??");
                throw new InvalidDataException("No valid Humidity found for yesterday");
            }

        }
        var floatValue = firstValidValue.Humidity;
        return floatValue;
    }
}
