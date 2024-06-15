using Azure.Identity;
using Azure.Storage.Blobs;
using Azure.Storage.Blobs.Models;
using Azure.Storage.Blobs.Specialized;
using Azure.Storage;
using Azure;

namespace app.Services;

public interface IFileService
{
    Task UploadFromFileAsync(BlobContainerClient containerClient, string localFilePath);
    Task<BlobContainerClient?> EnsureContainer(string containerName);
}

public class FileService : IFileService
{
    private readonly BlobServiceClient _blobServiceClient;
    public FileService(IConfiguration configuration){
        var _accountName = configuration.GetValue<string>("accountName");
        if(string.IsNullOrEmpty(_accountName)){
            throw new NullReferenceException("FileStorage:accountName cannot be NULL");
        }
        var _accountKey = configuration.GetValue<string>("accountKey");
        if(string.IsNullOrEmpty(_accountKey)) throw new NullReferenceException("FileStorage:accountKey cannot be NULL");

        _blobServiceClient = GetBlobServiceClient(_accountName, _accountKey);
    }
    private BlobServiceClient GetBlobServiceClient(string accountName, string accountKey)
    {
        StorageSharedKeyCredential sharedKeyCredential = new StorageSharedKeyCredential(accountName, accountKey);
        string blobUri = "https://" + accountName + ".blob.core.windows.net";

        var client = new BlobServiceClient(new Uri(blobUri), sharedKeyCredential);
        return client;
    }

    public async Task<BlobContainerClient?> EnsureContainer(string containerName)
    {
        var containerClient = _blobServiceClient.GetBlobContainerClient(containerName);

        try
        {
            // Create the container
            
            await containerClient.CreateIfNotExistsAsync();
            if (await containerClient.ExistsAsync())
            {
                Console.WriteLine("Created container {0}", containerClient.Name);
                return containerClient;
            }
        }
        catch (RequestFailedException e)
        {
            Console.WriteLine("HTTP error code {0}: {1}", e.Status, e.ErrorCode);
            Console.WriteLine(e.Message);
        }

        return containerClient;
    }
    public async Task UploadFromFileAsync(
                                BlobContainerClient containerClient,
                                string localFilePath)
    {

        string fileName = Path.GetFileName(localFilePath);
        BlobClient blobClient = containerClient.GetBlobClient(fileName);

        await blobClient.UploadAsync(localFilePath, true);
    }
}