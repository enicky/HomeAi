namespace Common.Models.Influx;

public record InfluxRecord{
    public DateTime Time { get; set; }
    public float Watt { get; set; }
    public double Humidity { get; set; }
    public double Pressure { get; set; }
    public double Temperature { get; set; }
}