FROM public.ecr.aws/lambda/python:3.8

RUN pip install numpy pandas xgboost sklearn

COPY ["loan_dv.bin", "loan_model.bin", "lambda_function.py", "./"]

CMD ["lambda_function.lambda_handler"]
