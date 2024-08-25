using Azure.Storage.Blobs;
using Azure.Storage;
using Azure;
using Microsoft.Extensions.Configuration;
using Common.Exceptions;

namespace Common.Services;

public interface IFileService
{
    Task UploadFromFileAsync(BlobContainerClient containerClient, string localFilePath);
    Task<BlobContainerClient?> EnsureContainer(string containerName);
}

public class FileService : IFileService
{
    private readonly BlobServiceClient _blobServiceClient;
    public FileService(IConfiguration configuration){
        Console.WriteLine($"accountn {configuration.GetValue<string>("FileStorage:accountName")}");
        var _accountName = configuration.GetValue<string>("FileStorage:accountName");
        if(string.IsNullOrEmpty(_accountName)){
            throw new AccountNameNullException("FileStorage:accountName cannot be NULL");
        }
        var _accountKey = configuration.GetValue<string>("accountKey");
        if(string.IsNullOrEmpty(_accountKey)) throw new AccountKeyNullException("FileStorage:accountKey cannot be NULL");

        _blobServiceClient = GetBlobServiceClient(_accountName, _accountKey);
    }
    private static BlobServiceClient GetBlobServiceClient(string accountName, string accountKey)
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