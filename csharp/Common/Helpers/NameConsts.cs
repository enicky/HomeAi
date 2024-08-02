namespace Common.Helpers;

public static class NameConsts{
    // INFLUX related consts
    public const string INFLUX_PUBSUB_NAME = "influx-data-pubsub";
    public const string INFLUX_RETRIEVE_DATA = "retrieve-data";
    public const string INFLUX_FINISHED_RETRIEVE_DATA = "finished-retrieve-data";



    // AI Training related Contst
    public const string AI_PUBSUB_NAME = "ai-pubsub";
    public const string AI_START_DOWNLOAD_DATA = "start-download-data";
    public const string AI_FINISHED_DOWNLOAD_DATA = "finished-download-data";
    public const string AI_START_TRAIN_MODEL = "start-train-model";
    public const string AI_FINISHED_TRAIN_MODEL = "finished-train-model";

}