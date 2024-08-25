using Azure.Storage.Blobs;

namespace Common.Factory;

public interface IBlobServiceClientFactory
{
    BlobServiceClient Create(string accountName, string accountKey);
}


