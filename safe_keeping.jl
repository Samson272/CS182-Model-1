#=
originals:
- Julia version:
- Author: shlomo
- Date: 2020-12-08
=#

using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPSimulators, Plots, QMDP, JLD, NPZ, DelimitedFiles

m = QuickPOMDP(
    states = ["healthy", "sick", "isolated", "removed"],
    actions = ["isolate", "visit family", "test", "medicate"],
    observations = ["fever", "no fever"],
    initialstate = SparseCat(["healthy", "sick", "isolated", "removed"], [1, 0, 0, 0]),
    discount = 0.95,



    transition = function (s, a)
        # Probability parameters:
        probability_to_get_removed_from_isolation = 0.025
        probability_to_be_okay_if_isolated = 0.325
        probability_to_need_more_isolated_if_isolated = 0.55
        probability_to_be_healthy_if_sick = 0.05

        probability_to_stay_sick_if_sick = 0.45
        probability_to_get_isolated_if_sick_without_a_test = 0.5
        probability_to_get_removed_if_sick = 0.05
        probability_to_get_sick_if_visited_family = 0.35
        probability_to_stay_healthy_if_visited_family = 0.15
        probability_to_be_isolated_if_visited_family = 0.5
        probability_to_be_removed_if_visited_family = 0.015

        probability_to_be_healed_if_medicated = 0.95
        probability_to_stay_sick_if_medicated = 0.05
        probability_to_need_isolation_if_medicated = 0
        probability_to_be_removed_if_medicated = 0


        if a == "isolate"
            return SparseCat(["healthy", "sick", "isolated"], [probability_to_be_okay_if_isolated, 0,
            probability_to_need_more_isolated_if_isolated,
            probability_to_get_removed_from_isolation])
        elseif s == "sick"
            return SparseCat(["healthy", "sick", "isolated", "removed"], [probability_to_be_healthy_if_sick,
            probability_to_stay_sick_if_sick,
            probability_to_get_isolated_if_sick_without_a_test, probability_to_get_removed_if_sick])
        elseif a == "visit family" #&& s == "healthy"
            return SparseCat(["healthy", "sick", "isolated", "removed"], [probability_to_stay_healthy_if_visited_family,
            probability_to_get_sick_if_visited_family,
            probability_to_be_isolated_if_visited_family, probability_to_be_removed_if_visited_family])
#         elseif a == "visit family"
#             return SparseCat(["healthy", "sick", "isolated", "remove"], [0, 0, 0, 1])
#         elseif a == "nothing"
#             return Uniform(["healthy", "sick"])
        elseif a == "test"
            return Deterministic(s)
        elseif a == "medicate" #&& s == "isolated"
            return SparseCat(["healthy", "sick", "isolated", "removed"], [probability_to_be_healed_if_medicated,
            probability_to_stay_sick_if_medicated,
            probability_to_need_isolation_if_medicated, probability_to_be_removed_if_medicated])
#         elseif a == "nothing"
#             return Deterministic(s)
        elseif s == "removed"
            return SparseCat(["healthy", "sick", "isolated", "removed"], [0, 0, 0, 1]) # RESET, terminal state.

        end
    end,

    observation = function (s, a, sp)

    probability_to_be_sick_if_fever = 0.35
    probability_to_be_isolated_if_fever = 0.25
    probability_to_be_removed_if_fever = 0.05
    probability_to_be_healthy_if_fever = 0
    probability_to_be_sick_or_removed_or_isolared_if_no_fever = 0
    probability_to_be_healthy_if_no_fever = 1

        if a == "test"
            if sp == "fever"
                return SparseCat(["healthy", "sick", "isolated", "removed"], [probability_to_be_healthy_if_fever,
                probability_to_be_sick_if_fever, probability_to_be_isolated_if_fever,
                probability_to_be_removed_if_fever]) # sparse categorical distribution
            else
                return SparseCat(["healthy", "sick", "isolated", "removed"], [probability_to_be_healthy_if_no_fever,
                probability_to_be_sick_or_removed_or_isolared_if_no_fever,
                probability_to_be_sick_or_removed_or_isolared_if_no_fever, probability_to_be_sick_or_removed_or_isolared_if_no_fever])
            end
        else # action is not 'test'
            return Deterministic(s)
        end
    end,

    reward = function (s, a)
        if a == "test"
            return -0.5
        elseif a == "visit family" && s == "healthy"
            return 1.5
        elseif a == "isolate"
            return -1.0
        elseif a == "medicate"  && s == "isolated"
            return -0.5
        elseif a == "nothing" && s == "isolated"
            return -0.5
        else
            return 0
        end
    end
)

global arr = [1.0]
function f(r)
     push!(arr,r)
end

solver = QMDPSolver()
policy = solve(solver, m)

rsum = 0.0
k = 30


global counter = 0
global quarantined = false
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=k)

#     if counter ==
#         break
#     end

    if s == "isolated"
        if quarantined == true
            a = "nothing"
            r = 0.0
            println("Step waited at quarantine, no reward earned. s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o, r: $r")
#             if o == "healthy"
#                 s = "healthy"
#             end
            global rsum += r
           global counter += 1
           global quarantined = false
           f(rsum)
           continue
        end
        quarantined = true
    end
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o, r: $r, total_reward: $rsum")
    global rsum += r
    #push!(arr, r)

    if s == "removed"
        println("Resident was removed from nursery")
        break
    end

    f(rsum)
end

println("Undiscounted reward was $rsum.")

println(arr)
writedlm( "rewardRunV.2.12.csv",  arr, ',')


# x = 1:size(arr); y = arr # 2 columns means two lines
# plot(x, y, title = "Two Lines", label = ["Line 1"], lw = 3)
