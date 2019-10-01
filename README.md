# StoryEndGen-pytorch
Pytorch implementation of StoryEndGen. Data preprocessed from https://github.com/JianGuanTHU/StoryEndGen.git.

### Dependency
- [Pattern](https://github.com/clips/pattern)
    - Install Pattern through PyPi repository  
      ```shell
      pip install pattern
      ```
    - To handle Pattern-3.6 [Issue](https://github.com/clips/pattern/issues/282) 
      with python 3.7, replace the following lines 608-609 of `[site-package dir]
      /Pattern-3.6-py3.7.egg/pattern/text/__init__.py`
      ```python
              yield line
        raise StopIteration
      ```  
      
      with
      
      ```python
        try:
            yield line
        except:
            StopIteration
      ```