from typing import *
from functools import *

class Solution:

    def hasValidPath(self, grid: List[List[str]]) -> bool:

        m, n = len(grid), len(grid[0])
        
        @cache
        def dfs(x: int, y: int, c: int) -> bool:

            if x == m - 1 and y == n - 1 and grid[x][y] == ')': 
                return c == 1  
 
            if grid[x][y] == '(':
                c += 1
            else:
                c -= 1

            return c >= 0 and (x < m - 1 and dfs(x + 1, y, c) or y < n - 1 and dfs(x, y + 1, c))
        
        return dfs(0, 0, 0)




if __name__ == "__main__":
    s = Solution()
    

    print(s.hasValidPath([["(","(","("],[")","(",")"],["(","(",")"],["(","(",")"]]))