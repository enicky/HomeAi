FROM --platform=$BUILDPLATFORM  mcr.microsoft.com/dotnet/sdk:8.0 AS build-env
WORKDIR /app

# Copy csproj and restore as distinct layers
COPY SchedulerService/*.csproj ./SchedulerService/
COPY Common/*.csproj ./Common/
COPY *.targets ./

RUN dotnet restore ./SchedulerService/SchedulerService.csproj

# Copy everything else and build
COPY SchedulerService ./SchedulerService
COPY Common ./Common
RUN ls -la 
RUN ls -la ./SchedulerService/*
RUN dotnet publish ./SchedulerService/SchedulerService.csproj -c Release -o out

# Build runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build-env /app/out .
EXPOSE 8080
EXPOSE 80
ENTRYPOINT ["dotnet", "SchedulerService.dll"]