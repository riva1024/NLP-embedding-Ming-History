# NLP-embedding-Ming-History
NLP作业/n
使用：/n
1.配置环境/n
pip install langchain-text-splitters/n
pip install --upgrade --force-reinstall langchain /n
2.运行：/n
python 1_build_index.py/n
很慢，可能要加载1个小时/n
python 2_search.py/n
可以通过修改代码中#YOUR_QUERY = 语句改变想查询的东西/n

tips:如果等不了一个小时，可以修改1中 chunk_size=150,增大这个值会更快，但chunk大了会返回一大段而且之后的结果很屎/n


