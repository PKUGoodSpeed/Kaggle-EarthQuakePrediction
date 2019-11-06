This folder contains scripts to facilitate debugging in portfolio list and
related apis in real environments i.e. dev, staging or prod.

## API Response Checker:

The script `api_response_checker.py` helps to reproduce the incorrect api
responses. It is initially designed for b/141691321. It works in three steps:

1.  It queries `SalesCrmTeam` table joined with `request_logs` to get all teams
    that we want to check.
2.  It fetches sampled api request messages from `request_logs` for the selected
    team ids.
3.  It sends rpc calls to the api server to get responses. It then checks and
    records all/incorrect requests and responses.

#### Getting a credential file: (The credential file is need to send rpc calls

to the api server.)

-   Run

```
while [ 1 ]; do echo refresh;
crm/greentea/be/auth/get_gaiamint.sh; echo done; sleep 240; done; refresh;
```

- Then get the credential file at `$(echo ~)/secure/gt_api_cred.txt`.

#### Example command:

```
blaze run //crm/greentea/be/scripts/gt_api:api_response_checker -- \
--server=/bns/vw/borg/vw/bns/grm-be-prod/engage-sales-crm-api-server-prod.server/3
\
--cred_file=$(echo ~)/secure/gt_api_cred.txt --limit_to=10 --env=prod \
--date_suffix=last3days --api_name=PortfolioService.List \
--in_process_apis=OpportunityService.List,GoogleProductService.List \
--wildcard_error_string=LOST \
--output_dir_cns=/cns/ok-d/home/username/opps_exp_data/ \
--temp_dir_local=/usr/local/google/home/username/workspace/rpc_experiments \
--save_all
```

This command checks the Portfolio List responses for up to 10 teams. The
response is supposed to be incorrect if it contains word "LOST". It meanwhile
checks the in process rpc calls OpportunityService.List and
GoogleProductService.List under the corresponding PortfolioService.List api
calls. Since `save_all` is enabled, all requests and responses will be recorded
in cns.

#### Flags:

```
  --api_name: Api full method name, e.g. PortfolioService.List.
      (default: 'PortfolioService.List')
  --cred_file: Credential file needed for sending rpc calls.
  --date_suffix: date suffix, e.g. last3days.
      (default: 'last3days')
  --env: F1 environment. Possible choices are prod, dev, staging.
      (default: 'prod')
  --extra_request_conditions: Extra conditions to apply to the query,
      E.g. 'AND user_id IN (1, 2, 3, 4, 5)'
  --in_process_apis: In process apis that also need to be checked.
      E.g. 'OpportunityService.List,GoogleProductService.List'
      (a comma separated list)
  --limit_to: Maximum number of teams to be checked.
      (an integer)
  --output_dir_cns: Output dir in CNS.
  --[no]save_all: Whether save all the reports. False means only save incorrect
      response.
      (default: 'false')
  --server: Api server address.
      (default: '/bns/vk/borg/vk/bns/grm-be-prod/api-server-prod.server/0')
  --team_ids: Ids of the selected teams whose api responses will be checked.
      If not set, all teams will be checked.
      (a comma separated list)
  --temp_dir_local: Temporary Local directory to save records.
  --wildcard_error_string: The regular expression indicating the api response
      is incorrect. E.g. 'LOST'.
      (default: '')
```
