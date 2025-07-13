namespace Common.Models.AI;

public class StartUploadModel : TraceableEvent
{
    public string  ModelPath { get; set; } = string.Empty;
    public DateTime TriggerMoment { get; set; } = DateTime.Now;
}
