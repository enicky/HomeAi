namespace Common.Models.AI;

public abstract class TraceableEvent
{
    public string TraceParent { get; set; } = string.Empty;
}

public class StartDownloadDataEvent : TraceableEvent {}
public class StartTrainModelEvent : TraceableEvent {}
