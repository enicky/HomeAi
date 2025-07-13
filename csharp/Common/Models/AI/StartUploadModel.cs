namespace Common.Models.AI;

public class StartUploadModel
{
    public string  ModelPath { get; set; } = string.Empty;
    public DateTime TriggerMoment { get; set; } = DateTime.Now;

    public string TraceParent {
        get;
        set;
    }
}
