#include <cstdio>
#include <algorithm>

int n,m,k;

const int maxn = 205;
const int maxk = 10;
long long dis[maxn][maxn];
const long long inf = 1e18;
int ach_cnt[maxk];
int ach[maxk][3];
long long dp[1<<(2*maxk)][maxk];

void s2arr(int s, int *arr){
    for (int i = 0; i < k; i++){
        arr[i] = s & 3;
        s >>= 2;
    }
}

void arr2s(const int *arr, int &s){
    s = 0;
    for (int i = k-1; i >= 0; i--){
        s <<= 2;
        s |= arr[i];
    }
}

long long dfs(int s, int i){
    int arr[maxk];
    s2arr(s,arr);
    long long res = inf;
    if (arr[i] == ach_cnt[i]){
        printf("error: dfs(%d,%d)\n",s,i);
        exit(0);
    }
    int s0 = s + (1<<(2*i));
    long long last;
    for (int j = 0; j < k; j++){
        if (arr[j] + (j == i) == ach_cnt[j]){
            continue;
        }
        if (dp[s0][j] == inf){
            last = dfs(s0, j);
        }else{
            last = dp[s0][j];
        }
        res = std::min(res,last+dis[ach[i][arr[i]]][ach[j][arr[j]+(j==i)]]);
    }
    // printf("dp[%d][%d] = %d\n",s,i,res);
    return dp[s][i] = res;
}

int main(){
    scanf("%d%d%d",&n,&m,&k);
    for (int i = 1; i <= n; i++){
        for (int j = 1; j <= n; j++){
            dis[i][j] = i == j ? 0 : inf;
        }
    }
    for (int i = 1; i <= m; i++){
        int u,v,w;
        scanf("%d%d%d",&u,&v,&w);
        dis[u][v] = dis[v][u] = w;
    }
    for (int i = 0; i < k; i++){
        scanf("%d", &ach_cnt[i]);
        for (int j = ach_cnt[i]-1; j >= 0; j--){
            scanf("%d",&ach[i][j]);
        }
    }
    for (int kk = 1; kk <= n; kk++){
        for (int i = 1; i <= n; i++){
            for (int j = 1; j <= n; j++){
                dis[i][j] = std::min(dis[i][j],dis[i][kk]+dis[kk][j]);
            }
        }
    }
    for (int i = 0; i < (1<<(2*k)); i++){
        for (int j = 0; j < k; j++){
            dp[i][j] = inf;
        }
    }
    for (int i = 0; i < k; i++){
        int s;
        arr2s(ach_cnt,s);
        dp[s-(1<<(2*i))][i] = dis[1][ach[i][ach_cnt[i]-1]];
    }
    long long ans = inf;
    for (int i = 0; i < k; i++){
        ans = std::min(ans,dfs(0, i));
    }
    printf("%lld",ans);
    return 0;
}