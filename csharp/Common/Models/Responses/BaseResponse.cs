using System.Text.Json.Serialization;

namespace Common.Models.Responses;

public abstract class BaseResponse
{
    [JsonPropertyName("success")]
    public bool Success { get; set; } = false;
}
