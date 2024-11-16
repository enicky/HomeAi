using System.Text.Json.Serialization;

namespace Common.Models.Responses;

public class TrainAiModelResponse : BaseResponse
{
    [JsonPropertyName("model_path")]
    public string ModelPath { get; set; } = string.Empty;
}
