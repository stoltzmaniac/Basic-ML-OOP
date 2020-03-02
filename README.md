# Basic-ML-OOP  

Scott Stoltzman  


As data scientists, many of us are used to procedural programming. This repo will correspond to a set of blog posts on my blog -> [Stoltzmaniac](https://stoltzmaniac.com). The goal is to make a basic library of machine learning objects in order to help us solve problems in an object oriented fashion. each part will live within its own subdirectory to keep it all contained. Each part will have its own `requirements.txt` so you must run your code from those directories to work.

For example, if you want to utilize this for the very first portion:  

```bash
> cd ./01-Regression/Single-Linear-Regression/Part-01
> pip install -r requirements.txt
> python run_me.py
```  

You will be able to pass extra parameters into your CLI, these will be discussed.

---

### 01-Regression/Single-Linear-Regression/Part-01    

Read the blog post: Stoltzmaniac

One of the most basic "machine learning" algorithims is linear regression. The term "machine learning" is used very loosely here due to the fact this algorithm is 100+ years old. However, because it is fitting data and is very simple to use, it makes for a great starting point.

Here, we look at setting up the basics for an object and utilize the CLI to pass data. This will not actually perform anything other than printing out the object but it gives us a place to start.