using Azure.Storage.Blobs;

namespace DataExporter.Services.Factory;

public interface IBlobServiceClientFactory
{
    BlobServiceClient Create(string accountName, string accountKey);
}


