'''
Ideally this file should be used to collect information from the database

For now:
client request -> Rate Limiter -> API Gatewway -> Load Balancer -> Services
This is how it will work for the asking a question like:
Run GPT vs Claude vs Gemini for connect four, 3 times

After that, will automatically show analytics, but to view past runs will also be an API service


View Analytics:
View postgres database for recently ran information, and dynamically create 3 different types of tables
Needs a baseline instance for the delta action table

Otherwise, entropy and heatmap are done just using information that just happened


'''