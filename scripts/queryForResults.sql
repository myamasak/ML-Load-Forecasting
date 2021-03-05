select TrialJobEvent.timestamp, MetricData.timestamp, MetricData.trialJobId, MetricData.data
from MetricData
inner join TrialJobEvent on TrialJobEvent.trialJobId = MetricData.trialJobId
where TrialJobEvent.event = 'WAITING'