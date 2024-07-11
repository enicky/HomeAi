
using System.Runtime.Serialization;
using System.Runtime.Serialization.DataContracts;
using System.Text.Json.Serialization;
using Common.Models.Influx;

namespace Common.Models.Responses;

[DataContract]
public class RetrieveDataResponse{

    // [DataMember]
    // [JsonPropertyName("value")]
    // public IEnumerable<InfluxRecord> Value { get;  set; } = new List<InfluxRecord>();
    [DataMember]
    [JsonPropertyName("generatedFileName")]
    public string GeneratedFileName { get; set; } = string.Empty;

    [JsonPropertyName("success")]
    public bool Success { get; set; }
}