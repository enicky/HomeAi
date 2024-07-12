using System.Text.Json.Serialization;

namespace Common.Models.Responses;

public class AiDownloadResponse
{
    [JsonPropertyName("success")]
    public bool Success { get; set; }
    [JsonPropertyName("canStartTraining")]
    public bool CanStartTraining { get; set; }
}
