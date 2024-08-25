using System.Diagnostics.CodeAnalysis;
using Azure.Storage;
using Azure.Storage.Blobs;

namespace DataExporter.Services.Factory;

[ExcludeFromCodeCoverage]
public class BlobServiceClientFactory : IBlobServiceClientFactory
{
    public BlobServiceClient Create(string accountName, string accountKey)
    {
        return GetBlobServiceClient(accountName, accountKey);
        
    }
    private static BlobServiceClient GetBlobServiceClient(string accountName, string accountKey)
    {
        StorageSharedKeyCredential sharedKeyCredential = new StorageSharedKeyCredential(accountName, accountKey);
        string blobUri = "https://" + accountName + ".blob.core.windows.net";

        var client = new BlobServiceClient(new Uri(blobUri), sharedKeyCredential);
        return client;
    }
}
