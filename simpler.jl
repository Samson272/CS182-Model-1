#=
originals:
- Julia version:
- Author: shlomo
- Date: 2020-12-08
=#


# last version works
using POMDPs, QuickPOMDPs, POMDPModelTools, POMDPSimulators, QMDP

m = QuickPOMDP(
    states = ["healthy", "sick", "isolated", "infected"],
    actions = ["isolate", "visit family", "test", "medicate"],
    observations = ["fever", "no fever"],
    initialstate = SparseCat(["healthy", "sick", "isolated", "infected"], [1, 0, 0, 0]),
    discount = 0.95,

    transition = function (s, a)
        if a == "isolate"
            return SparseCat(["healthy", "sick", "isolated"], [0, 0, 1.0]) # tiger stays behind the same door
        elseif a == "visit family" #&& s == "healthy"# a door is opened
            return SparseCat(["healthy", "sick", "isolated"], [0.05, 0.95, 0])
#         elseif a == "visit family"
#             return SparseCat(["healthy", "sick", "isolated", "remove"], [0, 0, 0, 1])
#         elseif a == "nothing"
#             return Uniform(["healthy", "sick"])
        elseif a == "test"
            return Deterministic(s)
        elseif a == "medicate"
            return SparseCat(["healthy", "sick", "isolated"], [1, 0, 0])
        end
    end,

    observation = function (s, a, sp)
        if a == "test"
            if sp == "fever"
                return SparseCat(["healthy", "sick", "isolated"], [0, 0.05, 0.95]) # sparse categorical distribution
            else
                return SparseCat(["sick", "healthy", "isolated"], [0.7, 0.3, 0])
            end
        else # action is not 'test'
            return Deterministic(s)
        end
    end,

    reward = function (s, a)
        if a == "test"
            return -1.0
        elseif a == "visit family" && s == "healthy"
            return 2.0
#         elseif a == "nothing" # the tiger was escaped
#             return 0
        elseif a == "isolate"
            return -5.0
        elseif a == "medicate"
            return -1.0
        else
            return 0
        end
    end
)

solver = QMDPSolver()
policy = solve(solver, m)

rsum = 0.0
for (s,b,a,o,r) in stepthrough(m, policy, "s,b,a,o,r", max_steps=10)
    println("s: $s, b: $([pdf(b,s) for s in states(m)]), a: $a, o: $o")
    global rsum += r
end
println("Undiscounted reward was $rsum.")

