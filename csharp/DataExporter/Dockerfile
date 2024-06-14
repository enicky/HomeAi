FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build-env
WORKDIR /app

# Copy csproj and restore as distinct layers
COPY DataExporter/*.csproj ./DataExporter/
COPY Common/*.csproj ./Common/

RUN dotnet restore ./DataExporter/DataExporter.csproj

# Copy everything else and build
COPY DataExporter ./DataExporter
COPY Common ./Common
RUN ls -la 
RUN ls -la ./DataExporter/*
RUN dotnet publish ./DataExporter/DataExporter.csproj -c Release -o out

# Build runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build-env /app/out .
EXPOSE 8080
EXPOSE 80
ENTRYPOINT ["dotnet", "DataExporter.dll"]