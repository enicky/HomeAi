
using System.Runtime.Serialization;
using System.Runtime.Serialization.DataContracts;
using System.Text.Json.Serialization;

namespace Common.Models.Responses;

[DataContract]
public class RetrieveDataResponse{

    [DataMember]
    [JsonPropertyName("generatedFileName")]
    public string GeneratedFileName { get; set; } = string.Empty;

    [JsonPropertyName("success")]
    public bool Success { get; set; }
    [JsonPropertyName("startaiprocess")]
    public bool StartAiProcess { get; set; }
}