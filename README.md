# NLP-embedding-Ming-History
NLP作业
使用：
1.配置环境
pip install langchain-text-splitters
pip install --upgrade --force-reinstall langchain 
2.运行：
python 1_build_index.py
很慢，可能要加载1个小时
python 2_search.py
可以通过修改代码中#YOUR_QUERY = 语句改变想查询的东西

tips:如果等不了一个小时，可以修改1中 chunk_size=150,增大这个值会更快，但chunk大了会返回一大段而且之后的结果很屎


