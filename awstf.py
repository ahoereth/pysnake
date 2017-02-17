#! /usr/bin/env python3
from os import environ
from argparse import ArgumentParser
from datetime import datetime, timedelta
from itertools import groupby

import numpy as np
import boto3


AMIS = {
    ('p2.xlarge', 'us-east-1'): 'ami-18438b0e',  # N. Virginia
    ('p2.xlarge', 'us-west-1'): 'ami-dadefebc',  # Ireland
}


useast1 = boto3.client('ec2', region_name='us-east-1')
euwest1 = boto3.client('ec2', region_name='eu-west-1')
clients = [useast1, euwest1]


def get_avg_price(instance_type, hours=5):
    prices = []
    for client in clients:
        zones = client.describe_availability_zones()
        zones = [zone['ZoneName'] for zone in zones['AvailabilityZones']]
        history = client.describe_spot_price_history(
            StartTime=datetime.today() - timedelta(hours=hours),
            EndTime=datetime.today(),
            InstanceTypes=[instance_type],
            ProductDescriptions=['Linux/UNIX'],
            Filters=[{'Name': 'availability-zone', 'Values': zones}],
        )
        history = history['SpotPriceHistory']
        grouper = lambda item: item['AvailabilityZone']
        for zone, items in groupby(sorted(history, key=grouper), key=grouper):
            price = np.mean([float(i['SpotPrice']) for i in items])
            prices.append((zone, price))
    return sorted(prices, key=lambda t: t[1])


command = '''\
docker-machine create \\
    --driver amazonec2 \\
    --amazonec2-region {region} \\
    --amazonec2-zone {zone} \\
    --amazonec2-ami {ami} \\
    --amazonec2-instance-type {instance_type} \\
    --amazonec2-security-group {security_group} \\
    --amazonec2-request-spot-instance \\
    --amazonec2-spot-price {price_max:.4f} \\
    {name}\
'''

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-t', '--instance-type', default='p2.xlarge')
    parser.add_argument('-sg', '--security-group', default='docker-machine')
    parser.add_argument('--max-price-on-top', default=.1, type=float)
    parser.add_argument('--hours', default=5, type=float)
    args = parser.parse_args()

    zone, price = get_avg_price(args.instance_type, hours=args.hours)[0]

    print('# Instances of type {instance_type} are cheapest in region {zone} '
          'with an average price of US${price:.4f} over the last {hours} hours.'
          .format(instance_type=args.instance_type, zone=zone, price=price,
                  hours=args.hours))
    print('# Issue the following command to launch a spot instance '
          'or call this script with eval $(SCRIPT). \n')
    print(command.format(
        region=zone[:-1],
        zone=zone[-1],
        instance_type=args.instance_type,
        security_group=args.security_group,
        price_max=price + args.max_price_on_top,
        ami=AMIS[(args.instance_type, zone[:-1])],
        name='aws42',
    ))
