#Capstone project 1 for sm_az_26

#### Capstone Project Experimenting on Fibra Monterrey Data

I analyzed daily data of Fibra Monterrey of Mexico City which is similar to US
REITs. Most of their properties are aligned for industrial purposes and in my
eyes the industry has potential for growth in the long term. They have been
slowly acquiring more industrial properties and recently acquired Fibra Macquarie.

Experimentation task (problem)

I started using quarterly data as I invest that way. Once you have quarterly
report data you analyze and make a decision. But this has some limitants which
simply put is the amount of data. The analysis can not fully cover up to a model
training due to the amount of data but you can improve the quality of the analysis
focusing on feature engineering.

So here is the approach instead of quarterly use daily data and customize available
technical indicators to fit other periods of time monthly, quarterly, your call.
In my case I chose quarter because that aligns with my investment practice.

Now using daily data and custom indicators data points can be enough to train a
set of models and identify rules that can signal a potential chance to invest.
To experiment with I chose the quarterly growth to make predictions and look out
for rules and apply them only when growth is higher than one.

This way the number of operations is low in how many times you buy/sell and you
can decide to use the amount of your choice if data does convince you.

#### System Requirements

  * python ">=3.12" [python.org](https://www.python.org/downloads/)
  * uv  ">=0.9.5"   [docs.astral.sh](https://docs.astral.sh/uv/getting-started/installation/)

#### Data

Data is downloaded from yfinance during execution as I mentioned early report
data is limited but it can prove useful for a much more simple task.

If you are interested take a look at their website:

  * [fibramty.com](https://www.fibramty.com/en/inversionistas)

#### Models

During the model training stage I realized that conservative parameters
sometimes perform the same as best estimators found. Here I highly recommend you
give yourself some time to review them as this can help reduce time while
maintaining performance.

#### Cloning

Create a folder to clone there

```bash
    mkdir sm_az_26_c1_cleall
  
    cd sm_az_26_c1_cleall
```

Clone project to sm_az_26_c1_cleall/

```bash
    git clone https://github.com/cleall/sm_az_26_c1_cleall
```

Navigate to project folder sm_az_26_c1_cleall

```bash
    cd sm_az_26_c1_cleall
```

Install python version for project using uv

```bash
    uv python install 3.12
```

Create venv using previously installed python version

```bash
    uv venv --python 3.12
```

Activate the environment

```bash
    source .venv/bin/activate
```

Install project dependencies

```bash
    uv sync --locked
```

#### Using main.py

Check:

  * All system dependencies have been satisfied

Once ready you can execute the following from the root of your uv environment:

```bash
    uv run main.py
```

This is going to go through data acquisition to simulation execution.

You are going to see messages indicating what step is being executed with some
of the output for reference. Most of the heavy lifting is on the notebook but
the essentials are kept in the python files. In any case you can review the
notebook for clarity.

Thats all you reached the end of the README file.

Thanks for reviewing my project good luck!
