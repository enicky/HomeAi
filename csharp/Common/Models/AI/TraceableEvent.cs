using System.Text.Json.Serialization;
namespace Common.Models.AI;

public abstract class TraceableEvent
{
    [JsonPropertyName("traceparent")]
    public string TraceParent { get; set; } = string.Empty;
}

public class StartDownloadDataEvent : TraceableEvent {}
public class StartTrainModelEvent : TraceableEvent {}
