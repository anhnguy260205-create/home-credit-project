import kagglehub

# Download latest version
path = kagglehub.competition_download('home-credit-default-risk')

print("Path to competition files:", path)
